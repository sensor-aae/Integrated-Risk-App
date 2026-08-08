# app/app.py
"""
Integrated Risk App — Streamlit interface.

UI RULE
-------
This file computes NO risk numbers. It collects inputs, calls `risklib`, and
visualises what comes back. Every statistic on screen is produced by the engine
and can be reproduced outside the app by constructing the same MarketRiskConfig.

If a calculation is needed and does not exist, it goes in `risklib/` — not here.
(An earlier version of this file carried its own inline Kupiec implementation,
which is exactly the drift this rule exists to prevent.)
"""

from pathlib import Path
import sys

# The repo root must be on the path before risklib can be imported, so the
# imports below deliberately sit after this line (flake8 E402).
sys.path.append(str(Path(__file__).resolve().parents[1]))

import io  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from risklib.credit import compute_el_table, summarize_el  # noqa: E402
from risklib.data import load_prices, to_returns  # noqa: E402
from risklib.market import (  # noqa: E402
    DEFAULT_DURATIONS,
    MarketRiskConfig,
    MarketRiskModel,
    backtest_var_fhs,
    backtest_var_historical,
    erc_weights,
    incremental_var,
    scenario_corr_bump_mc,
    scenario_covariance_scale,
    scenario_equities_shock,
    scenario_historical_replay,
    scenario_rates_bp,
    scenario_single_name,
    var_parametric_normal_parts,
)

METHOD_LABELS = {
    "Historical Simulation": "historical",
    "Parametric (Normal)": "parametric",
    "Monte Carlo": "monte_carlo",
    "Filtered Historical (GARCH)": "fhs",
}

RATE_SENSITIVE_HINT = ("TLT", "IEF", "AGG", "BND", "EDV", "ZROZ", "SHY", "TIP", "LQD", "HYG")


# ===========================================================================
# CACHE LAYER
#
# Streamlit re-executes this script top to bottom on every widget interaction.
# Without caching, nudging the alpha slider would re-parse the CSV, refit the
# model, re-run the rolling backtest and re-simulate 100,000 Monte Carlo paths.
#
# Two constraints shape the signatures below:
#   - The uploaded-file object is not stably hashable across reruns, so the raw
#     BYTES are passed instead.
#   - NumPy arrays are unhashable, so weights arrive as a tuple and are
#     converted inside.
# ===========================================================================

@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, ret_method: str, date_col: str):
    prices = load_prices(io.BytesIO(file_bytes), date_col=(date_col or None))
    returns = to_returns(prices, method=ret_method)
    return prices, returns


@st.cache_data(show_spinner=False)
def cached_risk(returns: pd.DataFrame, weights_t: tuple, cfg_kwargs: dict) -> dict:
    """
    Fit MarketRiskModel and return its full summary.

    The whole config is the cache key, so every parameter that changes a number
    also busts the cache. Previously only alpha/horizon/exposure/method were
    keyed, and the Monte Carlo and GARCH settings were not passed at all.
    """
    model = MarketRiskModel(returns, np.array(weights_t), MarketRiskConfig(**cfg_kwargs))
    model.fit()
    return model.summary()


@st.cache_data(show_spinner=False)
def cached_backtest_hist(returns: pd.DataFrame, weights_t: tuple, alpha: float, window: int) -> dict:
    return backtest_var_historical(returns, np.array(weights_t), alpha=alpha, window=window)


@st.cache_data(show_spinner=False)
def cached_backtest_fhs(returns: pd.DataFrame, weights_t: tuple, alpha: float,
                        window: int, alpha_g: float, beta_g: float) -> dict:
    return backtest_var_fhs(returns, np.array(weights_t), alpha=alpha, window=window,
                            alpha_g=alpha_g, beta_g=beta_g)


@st.cache_data(show_spinner=False)
def cached_decomp(returns: pd.DataFrame, weights_t: tuple, alpha: float,
                  horizon: int, exposure: float) -> dict:
    return var_parametric_normal_parts(returns, np.array(weights_t), alpha=alpha,
                                       horizon_days=horizon, exposure=exposure)


@st.cache_data(show_spinner=False)
def cached_incremental(returns: pd.DataFrame, weights_t: tuple, alpha: float,
                       horizon: int, exposure: float) -> dict:
    return incremental_var(returns, np.array(weights_t), alpha=alpha,
                           horizon_days=horizon, exposure=exposure)


@st.cache_data(show_spinner=False)
def cached_corr(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    r_slice = returns.tail(lookback) if len(returns) >= lookback else returns
    return r_slice.corr()


def pass_fail(p: float, level: float = 0.05) -> str:
    return "PASS" if p > level else "FAIL"


# ===========================================================================
# SIDEBAR — the single source of every parameter
#
# Each control appears exactly ONCE. Previously the Monte Carlo simulation
# count, seed and backtest window were declared both here and again inside
# individual tabs, so the same conceptual setting had two independent values
# and the sidebar copy silently governed nothing.
# ===========================================================================

st.set_page_config(page_title="Integrated Risk App", layout="wide")
st.title("Integrated Risk App")
st.caption("Market and credit risk measurement, validation and stress testing. "
           "All calculations are performed by `risklib`; this interface only displays them.")

st.sidebar.header("1 · Portfolio")
ret_method = st.sidebar.selectbox("Return type", ["log", "simple"], index=0)
weights_mode = st.sidebar.selectbox("Weights", ["Equal", "Manual by column name"])
alpha = st.sidebar.slider("Confidence (α)", 0.80, 0.999, 0.99, 0.001)
horizon = int(st.sidebar.number_input("Horizon (days)", 1, 30, 1))
exposure = float(st.sidebar.number_input("Exposure", 0.0, 1e12, 1_000_000.0, step=1000.0))

st.sidebar.header("2 · Method")
method_label = st.sidebar.radio("Method", list(METHOD_LABELS), index=0)
method = METHOD_LABELS[method_label]

st.sidebar.header("3 · Simulation")
st.sidebar.caption("Used by Monte Carlo VaR and every scenario that re-simulates.")
n_sims = int(st.sidebar.number_input("Simulations", 5_000, 1_000_000, 100_000, step=5_000))
seed = int(st.sidebar.number_input("Random seed", 0, 10_000, 42, step=1))
shrink_lambda = st.sidebar.slider("Covariance shrinkage (λ)", 0.0, 0.2, 0.01, 0.005)

st.sidebar.header("4 · GARCH")
fit_garch = st.sidebar.checkbox(
    "Estimate parameters by MLE",
    value=False,
    help="Off: variance targeting on the fixed α/β below. "
         "On: ω, α and β are estimated from the data by maximum likelihood.",
)
if fit_garch:
    alpha_g, beta_g = 0.05, 0.94   # retained for the FHS backtest, which uses fixed params
    st.sidebar.caption("Point estimates use MLE. The rolling FHS backtest keeps fixed "
                       "parameters — re-estimating at every step is a separate exercise.")
else:
    alpha_g = float(st.sidebar.number_input("GARCH α (ARCH)", 0.0, 0.5, 0.05, 0.01))
    beta_g = float(st.sidebar.number_input("GARCH β (GARCH)", 0.0, 0.999, 0.94, 0.01))
    if alpha_g + beta_g >= 1.0:
        st.sidebar.error(f"α + β = {alpha_g + beta_g:.3f} ≥ 1 — the process is "
                         "non-stationary. Reduce one of them.")

st.sidebar.header("5 · Backtest")
bt_window = int(st.sidebar.number_input("Rolling window (days)", 50, 1000, 250, step=10))


# ===========================================================================
# MARKET DATA
# ===========================================================================

st.markdown("### Market data")
st.caption("Upload a prices CSV: a date column plus one column per instrument.")
file = st.file_uploader("Prices CSV", type=["csv"], key="prices_csv")
date_col = st.text_input("Date column name (optional)", value="")

has_market = False
prices = returns = weights = None
summary = None

if file is not None:
    try:
        prices, returns = cached_load(file.read(), ret_method, date_col)
        has_market = True
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
else:
    st.info("No market CSV uploaded. The **Credit — Expected Loss** section below "
            "works independently and needs no price data.")

if has_market:
    st.subheader("Preview")
    c1, c2 = st.columns([3, 1])
    c1.dataframe(prices.tail(), use_container_width=True)
    c2.metric("Instruments", returns.shape[1])
    c2.metric("Return observations", len(returns))
    c2.caption(f"{returns.index.min():%Y-%m-%d} → {returns.index.max():%Y-%m-%d}")

    if weights_mode == "Equal":
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    else:
        st.write("Enter weights as `TICKER=WEIGHT, TICKER=WEIGHT`. They are normalised to sum to 1.")
        manual = st.text_input("Weights", value="", placeholder="SPY=0.6, TLT=0.4")
        wdict, bad = {}, []
        for part in (p.strip() for p in manual.split(",") if p.strip()):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    wdict[k.strip()] = float(v)
                except ValueError:
                    bad.append(part)
            else:
                bad.append(part)
        if bad:
            st.warning(f"Ignored unparseable entries: {', '.join(bad)}")
        unknown = set(wdict) - set(returns.columns)
        if unknown:
            st.warning(f"Not in the uploaded data, ignored: {', '.join(sorted(unknown))}")

        weights = np.array([wdict.get(c, 0.0) for c in returns.columns], dtype=float)
        if weights.sum() <= 0:
            st.warning("No valid weights supplied — falling back to equal weighting.")
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        else:
            weights = weights / weights.sum()

if has_market:
    weights_t = tuple(weights.tolist())
    cfg_kwargs = dict(
        alpha=alpha, method=method, horizon_days=horizon, exposure=exposure,
        window=bt_window, n_sims=n_sims, seed=seed, shrink_lambda=shrink_lambda,
        fit_garch=fit_garch, alpha_g=alpha_g, beta_g=beta_g,
    )


# ===========================================================================
# POINT RISK MEASURES
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("Point risk measures")
    try:
        summary = cached_risk(returns, weights_t, cfg_kwargs)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"VaR @ {alpha:.1%}, {horizon}d", f"{summary['VaR']:,.0f}")
        c2.metric("Expected Shortfall", f"{summary['ES']:,.0f}")
        c3.metric("ES / VaR", f"{summary['ES'] / summary['VaR']:.2f}"
                  if summary["VaR"] else "—")
        st.caption("Reported as positive loss amounts. The engine enforces VaR ≥ 0 and ES ≥ VaR.")

        with st.expander("Assumptions in force"):
            for a in summary["assumptions"]:
                st.write(f"- {a}")
            if "fit_info" in summary:
                fi = summary["fit_info"]
                st.markdown("**GARCH parameters**")
                src = fi.get("source", "—")
                st.write(f"Source: **{src}**")
                if "fallback_reason" in fi:
                    st.warning(f"MLE requested but fell back to fixed parameters: {fi['fallback_reason']}")
                if all(k in fi for k in ("omega", "alpha_g", "beta_g")):
                    st.write(f"ω = {fi['omega']:.3e}  |  α = {fi['alpha_g']:.4f}  |  "
                             f"β = {fi['beta_g']:.4f}  |  persistence = {fi['persistence']:.4f}")
                if "log_likelihood" in fi:
                    st.write(f"log-likelihood = {fi['log_likelihood']:,.1f}  |  "
                             f"AIC = {fi['aic']:,.1f}  |  BIC = {fi['bic']:,.1f}")
    except Exception as exc:
        st.error(f"Could not compute risk measures: {exc}")


# ===========================================================================
# BACKTESTING
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("Backtesting")

    if len(returns) <= bt_window:
        st.warning(f"Not enough history: {len(returns)} return observations against a "
                   f"{bt_window}-day window. Reduce the window or supply a longer series.")
    else:
        bt_kind = st.radio("Backtested model", ["Historical Simulation", "Filtered Historical (GARCH)"],
                           horizontal=True)
        if bt_kind == "Historical Simulation":
            bt = cached_backtest_hist(returns, weights_t, alpha, bt_window)
        else:
            bt = cached_backtest_fhs(returns, weights_t, alpha, bt_window, alpha_g, beta_g)

        left, right = st.columns([2, 1])

        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["r_p"].index, y=bt["r_p"].values,
                                     mode="lines", name="Portfolio return",
                                     line=dict(width=1)))
            fig.add_trace(go.Scatter(x=bt["VaR_threshold"].index, y=bt["VaR_threshold"].values,
                                     mode="lines", name=f"VaR threshold ({alpha:.1%})"))
            exc = (bt["exceptions"] == 1) & bt["VaR_threshold"].notna()
            fig.add_trace(go.Scatter(x=bt["r_p"].index[exc], y=bt["r_p"].values[exc],
                                     mode="markers", name="Exception",
                                     marker=dict(size=7, symbol="x")))
            fig.update_layout(height=430, xaxis_title="Date", yaxis_title="Return",
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Exceptions bunched into one region of the horizontal axis are exactly "
                       "what the independence test quantifies. The threshold at each date uses "
                       "only information available the day before.")

        with right:
            st.markdown("**Sample**")
            st.write(f"Window: **{bt['window']}** days")
            st.write(f"Out-of-sample points (T): **{bt['T']}**")
            st.write(f"Exceptions (x): **{bt['exceedances']}**")
            st.write(f"Hit rate: **{bt['hit_rate']:.4f}**  (expected {1 - alpha:.4f})")

            st.markdown("---")
            st.markdown("**① Kupiec POF** — unconditional coverage")
            st.caption("H₀: exception rate equals (1 − α)")
            st.write(f"LR = **{bt['kupiec_LR']:.3f}**, p = **{bt['kupiec_pvalue']:.4f}** "
                     f"→ **{pass_fail(bt['kupiec_pvalue'])}**")

            st.markdown("**② Christoffersen** — independence")
            st.caption("H₀: exceptions are serially independent")
            st.write(f"LR = **{bt['christoffersen_LR']:.3f}**, p = **{bt['christoffersen_pvalue']:.4f}** "
                     f"→ **{pass_fail(bt['christoffersen_pvalue'])}**")

            with st.expander("Transition matrix"):
                tr = bt["transitions"]
                st.write(f"n₀₀ = {tr['n00']}   n₀₁ = {tr['n01']}")
                st.write(f"n₁₀ = {tr['n10']}   n₁₁ = {tr['n11']}")
                st.write(f"π₀₁ = P(exc | no exc) = **{tr['pi_01']:.4f}**")
                st.write(f"π₁₁ = P(exc | exc)    = **{tr['pi_11']:.4f}**")
                st.caption("Under independence these are approximately equal. "
                           "π₁₁ > π₀₁ indicates clustering.")

            st.markdown("**③ Joint conditional coverage**")
            st.caption("H₀: correct frequency AND independence (LR_cc ~ χ²(2))")
            st.write(f"LR = **{bt['joint_LR']:.3f}**, p = **{bt['joint_pvalue']:.4f}** "
                     f"→ **{pass_fail(bt['joint_pvalue'])}**")

            st.markdown("---")
            failed = [name for name, key in
                      (("Kupiec", "kupiec_pvalue"), ("Christoffersen", "christoffersen_pvalue"),
                       ("Joint CC", "joint_pvalue"))
                      if bt[key] <= 0.05]
            if failed:
                st.error(f"Rejected at 5%: {', '.join(failed)}")
            else:
                st.success("All three tests pass at 5% significance.")

        bt_df = pd.DataFrame({
            "date": bt["r_p"].index,
            "return": bt["r_p"].values,
            "VaR_threshold": bt["VaR_threshold"].values,
            "exception": bt["exceptions"].values,
        })
        st.download_button("Download backtest series (CSV)",
                           bt_df.to_csv(index=False).encode(),
                           file_name=f"backtest_{bt['method']}_{int(alpha * 100)}pct_w{bt_window}.csv",
                           mime="text/csv")


# ===========================================================================
# CALIBRATION ACROSS CONFIDENCE LEVELS
# ===========================================================================

if has_market and len(returns) > bt_window:
    st.markdown("---")
    st.header("Calibration across confidence levels")
    st.caption("A model can be well calibrated at 95% and badly calibrated at 99% — the two "
               "sit in different parts of the tail. Testing across α is standard outcomes analysis.")

    with st.expander("Run multi-α calibration"):
        alphas = st.multiselect("Confidence levels", [0.90, 0.95, 0.975, 0.99, 0.995],
                                default=[0.95, 0.975, 0.99])
        include_fhs = st.checkbox("Include the FHS backtest for comparison", value=True)

        if st.button("Run calibration") and alphas:
            rows = []
            for a in sorted(alphas):
                h = cached_backtest_hist(returns, weights_t, a, bt_window)
                row = {
                    "alpha": a,
                    "expected_%": (1 - a) * 100,
                    "HS_T": h["T"], "HS_exceptions": h["exceedances"],
                    "HS_hit_%": h["hit_rate"] * 100,
                    "HS_kupiec_p": h["kupiec_pvalue"],
                    "HS_christoffersen_p": h["christoffersen_pvalue"],
                    "HS_joint_p": h["joint_pvalue"],
                }
                if include_fhs:
                    f = cached_backtest_fhs(returns, weights_t, a, bt_window, alpha_g, beta_g)
                    row.update({
                        "FHS_exceptions": f["exceedances"],
                        "FHS_hit_%": f["hit_rate"] * 100,
                        "FHS_kupiec_p": f["kupiec_pvalue"],
                        "FHS_christoffersen_p": f["christoffersen_pvalue"],
                        "FHS_joint_p": f["joint_pvalue"],
                    })
                rows.append(row)

            calib = pd.DataFrame(rows)
            st.dataframe(calib.style.format({
                c: "{:.4f}" for c in calib.columns if c.endswith("_p")
            }), use_container_width=True)

            figc = go.Figure()
            labels = [f"{a:.1%}" for a in calib["alpha"]]
            figc.add_trace(go.Bar(x=labels, y=calib["expected_%"], name="Expected %"))
            figc.add_trace(go.Bar(x=labels, y=calib["HS_hit_%"], name="Historical hit %"))
            if "FHS_hit_%" in calib:
                figc.add_trace(go.Bar(x=labels, y=calib["FHS_hit_%"], name="FHS hit %"))
            figc.update_layout(barmode="group", height=360,
                               xaxis_title="Confidence level", yaxis_title="Percent")
            st.plotly_chart(figc, use_container_width=True)

            st.download_button("Download calibration (CSV)",
                               calib.to_csv(index=False).encode(),
                               file_name=f"calibration_w{bt_window}.csv", mime="text/csv")


# ===========================================================================
# ANALYTICS
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("Analytics")
    tab_corr, tab_decomp = st.tabs(["Correlation", "VaR decomposition"])

    with tab_corr:
        lookback = int(st.number_input("Lookback (days)", 50, 5000, 250, step=10, key="corr_lb"))
        corr = cached_corr(returns, lookback)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, title=f"Correlation — last {lookback} days"),
            use_container_width=True)
        st.caption("The scale is fixed to [−1, 1] so colour is comparable across datasets.")

    with tab_decomp:
        parts = cached_decomp(returns, weights_t, alpha, horizon, exposure)
        tickers = list(returns.columns)

        df_parts = pd.DataFrame({
            "Instrument": tickers,
            "Weight": parts["w"],
            "Marginal VaR": parts["mVaR"],
            "Component VaR": parts["cVaR"],
            "% of VaR": parts["pContrib"] * 100,
        })
        st.dataframe(df_parts.style.format({
            "Weight": "{:.2%}", "Marginal VaR": "{:,.0f}",
            "Component VaR": "{:,.0f}", "% of VaR": "{:.2f}%",
        }), use_container_width=True)

        st.plotly_chart(px.bar(df_parts, x="Instrument", y="Component VaR",
                               title="Component VaR"), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio VaR (Normal)", f"{parts['VaR']:,.0f}")
        c2.metric("μ (portfolio, horizon)", f"{parts['mu_p']:.6f}")
        c3.metric("σ (portfolio, horizon)", f"{parts['sigma_p']:.6f}")
        st.caption("Component VaR sums to portfolio VaR **exactly** — VaR is homogeneous of "
                   "degree 1 in the weights, so Euler's theorem applies with equality. "
                   "Marginal VaR is the sensitivity to a small increase in a weight.")

        inc = cached_incremental(returns, weights_t, alpha, horizon, exposure)
        st.markdown("**Incremental VaR**")
        st.dataframe(pd.DataFrame({
            "Instrument": tickers,
            "Incremental VaR": inc["iVaR"],
            "Component VaR": inc["cVaR"],
        }).style.format({"Incremental VaR": "{:,.0f}", "Component VaR": "{:,.0f}"}),
            use_container_width=True)
        st.caption("Incremental VaR is a true recomputation — portfolio VaR minus the VaR of "
                   "the portfolio with that position removed and the rest renormalised. It "
                   "converges to component VaR only for small positions.")


# ===========================================================================
# WHAT-IF WEIGHTS
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("What-if: adjust weights")

    tickers = list(returns.columns)
    cols = st.columns(min(4, max(2, len(tickers))))
    new_w = []
    for i, t in enumerate(tickers):
        with cols[i % len(cols)]:
            new_w.append(st.slider(t, 0.0, 1.0, float(weights[i]), 0.01, key=f"w_{t}"))
    new_w = np.array(new_w, dtype=float)

    if new_w.sum() <= 0:
        st.warning("All weights are zero — showing the current portfolio instead.")
        new_w = weights.copy()
    new_w = new_w / new_w.sum()

    curr = cached_decomp(returns, weights_t, alpha, horizon, exposure)
    what = cached_decomp(returns, tuple(new_w.tolist()), alpha, horizon, exposure)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current VaR (Normal)", f"{curr['VaR']:,.0f}")
    c2.metric("What-if VaR (Normal)", f"{what['VaR']:,.0f}")
    c3.metric("Change", f"{what['VaR'] - curr['VaR']:,.0f}",
              delta=f"{(what['VaR'] / curr['VaR'] - 1) * 100:.2f}%" if curr["VaR"] else None,
              delta_color="inverse")

    st.dataframe(pd.DataFrame({
        "Instrument": tickers,
        "Weight (current)": curr["w"],
        "Weight (what-if)": new_w,
        "cVaR (current)": curr["cVaR"],
        "cVaR (what-if)": what["cVaR"],
        "% of VaR (current)": curr["pContrib"] * 100,
        "% of VaR (what-if)": what["pContrib"] * 100,
    }).style.format({
        "Weight (current)": "{:.2%}", "Weight (what-if)": "{:.2%}",
        "cVaR (current)": "{:,.0f}", "cVaR (what-if)": "{:,.0f}",
        "% of VaR (current)": "{:.2f}%", "% of VaR (what-if)": "{:.2f}%",
    }), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=tickers, y=curr["cVaR"]))
    fig.add_trace(go.Bar(name="What-if", x=tickers, y=what["cVaR"]))
    fig.update_layout(barmode="group", title="Component VaR", height=380)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# STRESS TESTING
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("Stress testing")
    st.caption(f"Scenarios re-simulate with {n_sims:,} paths and seed {seed} "
               "(set in the sidebar), so base and stressed figures differ by the "
               "scenario alone, not by Monte Carlo noise.")

    t_single, t_eq, t_rates, t_cov, t_corr, t_hist = st.tabs(
        ["Single name", "Equity shock", "Rate shock", "Volatility", "Correlation", "Historical replay"])

    tickers = list(returns.columns)

    with t_single:
        st.write("Apply one-day shocks in return space. Instruments not shocked are held flat.")
        picked = st.multiselect("Instruments to shock", tickers, default=tickers[:1])
        shocks = {}
        if picked:
            cols = st.columns(min(3, len(picked)))
            for i, t in enumerate(picked):
                with cols[i % len(cols)]:
                    shocks[t] = st.number_input(f"{t} shock", value=-0.10, step=0.01,
                                                format="%.4f", key=f"sn_{t}")
        if st.button("Run single-name shock") and shocks:
            res = scenario_single_name(returns, weights, shocks, exposure=exposure)
            st.metric("Scenario loss", f"{res['loss']:,.0f}")
            st.dataframe(pd.DataFrame({"Instrument": tickers,
                                       "Shock": res["shock_vector"].values,
                                       "Weight": weights}),
                         use_container_width=True)

    with t_eq:
        eqs = st.multiselect("Equity instruments", tickers,
                             default=[c for c in tickers if c not in RATE_SENSITIVE_HINT])
        eq_shock = st.number_input("Shock (return)", -1.0, 1.0, -0.20, 0.01)
        if st.button("Run equity shock"):
            loss = scenario_equities_shock(returns, weights, eqs,
                                           shock=float(eq_shock), exposure=exposure)
            st.metric("Scenario loss", f"{loss:,.0f}")

    with t_rates:
        st.write("Parallel rate shock via the duration approximation, ΔP/P ≈ −D × Δy.")
        bp = st.number_input("Rate move (bp)", -1000, 1000, 200, 25)
        st.caption("Durations default to 0 for anything not recognised as rate-sensitive. "
                   "A parallel rate shock should not move an equity position.")
        dur_inputs = {}
        cols = st.columns(min(4, max(2, len(tickers))))
        for i, c in enumerate(tickers):
            with cols[i % len(cols)]:
                dur_inputs[c] = st.number_input(f"{c} duration", 0.0, 40.0,
                                                float(DEFAULT_DURATIONS.get(c, 0.0)),
                                                step=0.5, key=f"dur_{c}")
        if st.button("Run rate shock"):
            res = scenario_rates_bp(returns, weights, durations=dur_inputs,
                                    bp=float(bp), exposure=exposure)
            st.metric("Scenario loss", f"{res['loss']:,.0f}")
            st.caption("Linear approximation — convexity is ignored, which overstates the "
                       "loss for large moves on long-duration instruments.")

    with t_cov:
        st.write("Scale the covariance matrix and re-simulate.")
        scale = st.slider("Covariance scale (×)", 0.5, 5.0, 2.0, 0.1)
        st.caption(f"Covariance ×{scale:.1f} means volatility ×{np.sqrt(scale):.2f}; "
                   "correlations are unchanged.")
        if st.button("Run volatility stress"):
            res = scenario_covariance_scale(returns, weights, scale=float(scale), alpha=alpha,
                                            horizon_days=horizon, exposure=exposure,
                                            n_sims=n_sims, seed=seed)
            c1, c2, c3 = st.columns(3)
            c1.metric("VaR (base)", f"{res['base']['VaR']:,.0f}")
            c2.metric("VaR (stressed)", f"{res['stressed']['VaR']:,.0f}")
            c3.metric("Change", f"{res['delta_VaR']:,.0f}")
            st.caption(f"ES {res['base']['ES']:,.0f} → {res['stressed']['ES']:,.0f}")

    with t_corr:
        st.write("Bump off-diagonal correlations, hold volatilities constant, re-simulate.")
        st.caption("Isolates the diversification-breakdown channel: in a crisis correlations "
                   "converge toward 1 and a diversified book turns out to be one bet.")
        corr_bump = st.slider("Correlation bump (%)", 0, 300, 50, 5)
        if st.button("Run correlation stress"):
            res = scenario_corr_bump_mc(returns, weights, alpha=alpha, horizon_days=horizon,
                                        exposure=exposure, corr_bump_pct=float(corr_bump),
                                        n_sims=n_sims, seed=seed)
            c1, c2, c3 = st.columns(3)
            c1.metric("VaR (base)", f"{res['base']['VaR']:,.0f}")
            c2.metric("VaR (stressed)", f"{res['stressed']['VaR']:,.0f}")
            c3.metric("Change", f"{res['delta_VaR']:,.0f}")
            if res["psd_adjusted"]:
                st.warning("The bumped correlation matrix was not positive semi-definite and "
                           "was projected back onto the PSD cone. At this magnitude the "
                           "scenario is straining the correlation structure.")

    with t_hist:
        st.write("Re-run the current portfolio through a past statistical regime.")
        min_d, max_d = returns.index.min().date(), returns.index.max().date()
        c1, c2 = st.columns(2)
        start = c1.date_input("Window start", min_d, min_value=min_d, max_value=max_d)
        end = c2.date_input("Window end", max_d, min_value=min_d, max_value=max_d)
        if st.button("Run historical replay"):
            try:
                res = scenario_historical_replay(returns, weights, str(start), str(end),
                                                 alpha=alpha, horizon_days=horizon,
                                                 exposure=exposure, n_sims=n_sims, seed=seed)
                c1, c2, c3 = st.columns(3)
                c1.metric("VaR (replayed regime)", f"{res['VaR']:,.0f}")
                c2.metric("ES (replayed regime)", f"{res['ES']:,.0f}")
                c3.metric("Observations used", res["n_obs"])
            except Exception as exc:
                st.error(str(exc))


# ===========================================================================
# RISK BUDGETING
# ===========================================================================

if has_market:
    st.markdown("---")
    st.header("Risk budgeting — Equal Risk Contribution")

    with st.expander("Compute ERC weights"):
        c1, c2 = st.columns(2)
        erc_min = c1.number_input("Minimum weight", 0.0, 1.0, 0.0, 0.01)
        erc_max = c2.number_input("Maximum weight", 0.0, 1.0, 1.0, 0.01)
        erc_step = st.slider("Update damping", 0.1, 1.0, 0.5, 0.1)
        init_current = st.checkbox("Initialise from current weights", value=True)

        n_assets = returns.shape[1]
        if n_assets * erc_max < 1.0 or n_assets * erc_min > 1.0:
            st.error(f"Bounds are infeasible for {n_assets} instruments: weights cannot "
                     f"sum to 1 with min={erc_min} and max={erc_max}.")
        elif st.button("Run ERC"):
            try:
                w_erc, info = erc_weights(
                    returns, horizon_days=horizon,
                    init=weights if init_current else None,
                    min_w=float(erc_min), max_w=float(erc_max), step=float(erc_step),
                )
                p_curr = cached_decomp(returns, weights_t, alpha, horizon, exposure)
                p_erc = cached_decomp(returns, tuple(w_erc.tolist()), alpha, horizon, exposure)

                c1, c2 = st.columns(2)
                c1.metric("VaR (current)", f"{p_curr['VaR']:,.0f}")
                c2.metric("VaR (ERC)", f"{p_erc['VaR']:,.0f}")

                df_erc = pd.DataFrame({
                    "Instrument": list(returns.columns),
                    "Weight (current)": p_curr["w"],
                    "Weight (ERC)": w_erc,
                    "% of VaR (current)": p_curr["pContrib"] * 100,
                    "% of VaR (ERC)": p_erc["pContrib"] * 100,
                })
                st.dataframe(df_erc.style.format({
                    "Weight (current)": "{:.2%}", "Weight (ERC)": "{:.2%}",
                    "% of VaR (current)": "{:.2f}%", "% of VaR (ERC)": "{:.2f}%",
                }), use_container_width=True)

                st.caption(
                    f"{'Converged' if info['converged'] else 'Stopped'} after {info['iter']} "
                    f"iterations; risk-contribution dispersion {info['rc_dispersion']:.2e}. "
                    "Contributions are equalised in variance space; percentage-of-VaR figures "
                    "additionally carry the drift term, so they are close but not identical."
                )
                st.download_button("Download ERC weights (CSV)",
                                   df_erc[["Instrument", "Weight (ERC)"]].to_csv(index=False).encode(),
                                   file_name="erc_weights.csv", mime="text/csv")
            except ValueError as exc:
                st.error(str(exc))


# ===========================================================================
# CREDIT — EXPECTED LOSS
# ===========================================================================

st.markdown("---")
st.header("Credit — Expected Loss")
st.write("Upload an exposures CSV with **PD**, **LGD** and **EAD** columns "
         "(case-insensitive, common aliases accepted). A segment, rating or grade "
         "column is detected automatically and used for grouping.")

credit_file = st.file_uploader("Exposures CSV", type=["csv"], key="credit_csv")

with st.expander("Scenario shocks"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pd_mult = st.number_input("PD multiplier (×)", 0.0, 100.0, 1.0, 0.05)
        pd_add_bps = st.number_input("PD additive (bp)", -5000, 5000, 0, 25)
    with c2:
        lgd_mult = st.number_input("LGD multiplier (×)", 0.0, 100.0, 1.0, 0.05)
        lgd_add_pct = st.number_input("LGD additive (pp)", -100, 100, 0, 1)
    with c3:
        ead_mult = st.number_input("EAD multiplier (×)", 0.0, 100.0, 1.0, 0.05)
    st.caption("Multiplicative shocks represent proportional deterioration; additive shocks "
               "represent a uniform level shift. Regulatory narratives use both.")

if credit_file is not None:
    try:
        cdf = pd.read_csv(credit_file)
        df_el, seg_col = compute_el_table(
            cdf, pd_mult=float(pd_mult), pd_add_bps=float(pd_add_bps),
            lgd_mult=float(lgd_mult), lgd_add_pct=float(lgd_add_pct),
            ead_mult=float(ead_mult))
        grp, totals = summarize_el(df_el, seg_col)

        quality = df_el.attrs.get("data_quality", {})
        issues = {k: v for k, v in quality.items() if v}
        if issues:
            st.warning("Data quality: " + "; ".join(
                f"{v} row(s) {k.replace('_', ' ')}" for k, v in issues.items()))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Facilities", f"{int(totals['facilities']):,}")
        c2.metric("Total EAD", f"{totals['total_EAD']:,.0f}")
        c3.metric("Total EL", f"{totals['total_EL']:,.0f}")
        c4.metric("EL / EAD", f"{totals['EL_pct_of_EAD'] * 100:.2f}%")
        st.caption("Portfolio EL/EAD is exposure-weighted (total EL ÷ total EAD), consistent "
                   "with the grouped subtotals below.")

        if not grp.empty:
            st.subheader(f"By {seg_col}")
            st.dataframe(grp.style.format({
                "total_EAD": "{:,.0f}", "total_EL": "{:,.0f}",
                "avg_PD": "{:.4f}", "avg_LGD": "{:.4f}",
                "EL_pct_of_EAD": "{:.4%}",
            }), use_container_width=True)
            st.plotly_chart(px.bar(grp, x=seg_col, y="total_EL", title="Expected Loss by segment"),
                            use_container_width=True)

        st.subheader("Per-facility detail")
        st.dataframe(df_el, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.download_button("Download detailed EL (CSV)",
                           df_el.to_csv(index=False).encode(),
                           file_name="credit_el_detailed.csv", mime="text/csv")
        if not grp.empty:
            c2.download_button("Download grouped summary (CSV)",
                               grp.to_csv(index=False).encode(),
                               file_name="credit_el_grouped.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Could not process the exposures file: {exc}")
else:
    st.info("Upload an exposures CSV to compute Expected Loss. No market data required.")


# ===========================================================================
# REPORT EXPORT
# ===========================================================================

if has_market and summary is not None:
    st.markdown("---")
    st.header("Report export")

    parts_now = cached_decomp(returns, weights_t, alpha, horizon, exposure)
    tickers = list(returns.columns)

    lines = [
        "# Risk Report", "",
        f"**Method:** {method_label}",
        f"**Confidence (α):** {alpha:.3f}  |  **Horizon:** {horizon}d  |  "
        f"**Exposure:** {exposure:,.0f}",
        f"**Sample:** {len(returns):,} observations, "
        f"{returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}",
        "", "## Portfolio", "", "| Instrument | Weight |", "|---|---:|",
    ]
    lines += [f"| {t} | {w:.4f} |" for t, w in zip(tickers, parts_now["w"])]

    lines += ["", "## Point risk measures", "",
              f"- VaR: **{summary['VaR']:,.0f}**",
              f"- ES: **{summary['ES']:,.0f}**",
              "", "### Assumptions", ""]
    lines += [f"- {a}" for a in summary["assumptions"]]

    lines += ["", "## VaR decomposition (Normal, Euler allocation)", "",
              "| Instrument | Weight | Component VaR | % of VaR |", "|---|---:|---:|---:|"]
    lines += [f"| {t} | {w:.4f} | {c:,.0f} | {p * 100:.2f}% |"
              for t, w, c, p in zip(tickers, parts_now["w"],
                                    parts_now["cVaR"], parts_now["pContrib"])]

    if len(returns) > bt_window:
        b = cached_backtest_hist(returns, weights_t, alpha, bt_window)
        lines += ["", "## Backtest — rolling historical VaR", "",
                  f"- Window: {b['window']}d  |  OOS observations: {b['T']}  |  "
                  f"Exceptions: {b['exceedances']}  |  Hit rate: {b['hit_rate']:.4f} "
                  f"(expected {1 - alpha:.4f})",
                  f"- Kupiec: LR = {b['kupiec_LR']:.3f}, p = {b['kupiec_pvalue']:.4f} "
                  f"({pass_fail(b['kupiec_pvalue'])})",
                  f"- Christoffersen: LR = {b['christoffersen_LR']:.3f}, "
                  f"p = {b['christoffersen_pvalue']:.4f} ({pass_fail(b['christoffersen_pvalue'])})",
                  f"- Joint CC: LR = {b['joint_LR']:.3f}, p = {b['joint_pvalue']:.4f} "
                  f"({pass_fail(b['joint_pvalue'])})"]

    c1, c2 = st.columns(2)
    c1.download_button("Download report (Markdown)", "\n".join(lines).encode(),
                       file_name="risk_report.md", mime="text/markdown")
    c2.download_button("Download VaR decomposition (CSV)",
                       pd.DataFrame({
                           "Instrument": tickers, "Weight": parts_now["w"],
                           "Component VaR": parts_now["cVaR"],
                           "Percent of VaR": parts_now["pContrib"],
                       }).to_csv(index=False).encode(),
                       file_name="var_decomposition.csv", mime="text/csv")
