"""Streamlit dashboard for the Kalman-pairs-trading walkthrough.

Run with::

    streamlit run dashboard/app.py

Sliders let you change ``alpha`` (smoothness vs adaptivity), the
z-score window, and the entry threshold, then re-run all three
estimators (rolling LS, basic Kalman, Kalman with momentum) and watch
how the hedge ratio, spread, signal and cumulative return change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalman_pairs import (  # noqa: E402
    backtest_pairs,
    kalman_basic,
    kalman_momentum,
    kalman_partial_coint,
    load_pair,
    rolling_ls_hedge,
    spread_zscore,
)


st.set_page_config(
    page_title="Kalman Pairs Trading Lab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Kalman filtering for pairs trading — interactive lab")
st.caption(
    "Section 15.6 of Palomar (2024). Move the sliders to see how the hedge "
    "ratio, spread and P&L change. Compare rolling least squares against two "
    "Kalman variants."
)


# ----- Sidebar controls ----------------------------------------------------

with st.sidebar:
    st.header("Data")
    pair = st.selectbox(
        "Pair",
        options=["EWA-EWC", "KO-PEP"],
        index=0,
        help="EWA-EWC is stable; KO-PEP has a regime shift in early 2020.",
    )
    ticker_a, ticker_b = pair.split("-")
    start = st.date_input("Start", value=pd.Timestamp("2013-01-01")).isoformat()
    end = st.date_input("End", value=pd.Timestamp("2022-12-31")).isoformat()

    st.divider()
    st.header("Kalman parameters")
    alpha_basic = st.select_slider(
        "alpha (basic)",
        options=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-5,
        format_func=lambda v: f"{v:.0e}",
        help="Ratio of hidden-state variability to spread noise. Smaller → smoother.",
    )
    alpha_mom = st.select_slider(
        "alpha (momentum)",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        value=1e-6,
        format_func=lambda v: f"{v:.0e}",
    )
    use_partial = st.checkbox("Also show partial-cointegration model (15.5)", value=False)
    rho = st.slider("rho (AR(1) residual)", min_value=0.0, max_value=0.99, value=0.90, step=0.01,
                    help="Only used by the partial-cointegration model.") if use_partial else 0.9
    t_ls = st.slider("LS warm-up window (days)", min_value=60, max_value=504, value=252, step=10)

    st.divider()
    st.header("Strategy")
    ls_window = st.slider("Rolling LS window (days)", min_value=60, max_value=756, value=504, step=20)
    zwindow = st.slider("z-score window (days)", min_value=20, max_value=252, value=126, step=10)
    s0 = st.slider("Entry threshold s0", min_value=0.5, max_value=3.0, value=1.0, step=0.1)


# ----- Compute ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_prices(ticker_a: str, ticker_b: str, start: str, end: str) -> pd.DataFrame:
    return load_pair(ticker_a, ticker_b, start=start, end=end)


@st.cache_data(show_spinner=False)
def run_models(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    alpha_basic: float,
    alpha_mom: float,
    rho: float,
    t_ls: int,
    ls_window: int,
    use_partial: bool,
) -> dict:
    y1 = prices[ticker_a]
    y2 = prices[ticker_b]
    ls = rolling_ls_hedge(y1, y2, window=ls_window)
    kb = kalman_basic(y1, y2, alpha=alpha_basic, t_ls=t_ls)
    km = kalman_momentum(y1, y2, alpha=alpha_mom, t_ls=t_ls)

    def normalized_spread(mu, gamma):
        return (y1 - gamma * y2 - mu) / (1.0 + gamma)

    out = {
        "y1": y1,
        "y2": y2,
        "gamma": {
            "Rolling LS": ls["gamma"],
            "Kalman basic": kb.states_filt["gamma"],
            "Kalman + momentum": km.states_filt["gamma"],
        },
        "mu": {
            "Rolling LS": ls["mu"],
            "Kalman basic": kb.states_filt["mu"],
            "Kalman + momentum": km.states_filt["mu"],
        },
        "spread": {
            "Rolling LS": normalized_spread(ls["mu"], ls["gamma"]),
            "Kalman basic": kb.spread,
            "Kalman + momentum": km.spread,
        },
    }
    if use_partial:
        kp = kalman_partial_coint(y1, y2, rho=rho, alpha=alpha_basic, t_ls=t_ls)
        out["gamma"]["Kalman partial coint"] = kp.states_filt["gamma"]
        out["mu"]["Kalman partial coint"] = kp.states_filt["mu"]
        out["spread"]["Kalman partial coint"] = kp.spread
        out["epsilon"] = kp.states_filt["epsilon"]
    return out


with st.spinner("Loading data..."):
    prices = get_prices(ticker_a, ticker_b, start, end)

source = "real Yahoo data" if (prices.index[0].year >= 2013 and len(prices) > 1500 and prices[ticker_a].std() > 1) else "data"
st.caption(f"Loaded {len(prices)} rows for **{pair}** "
           f"from {prices.index[0].date()} to {prices.index[-1].date()}.")

with st.spinner("Running estimators..."):
    res = run_models(prices, ticker_a, ticker_b, alpha_basic, alpha_mom, rho, t_ls, ls_window, use_partial)


# ----- Backtest ----------------------------------------------------------

backtests = {name: backtest_pairs(s, window=zwindow, s0=s0) for name, s in res["spread"].items()}

metrics = pd.DataFrame({
    name: {
        "final P&L": bt["cumulative_return"].iloc[-1],
        "max drawdown": (bt["cumulative_return"] - bt["cumulative_return"].cummax()).min(),
        "trades": int((bt["signal"].diff().abs() > 0).sum()),
        "% time in market": float((bt["signal"] != 0).mean()) * 100,
    } for name, bt in backtests.items()
}).T

cols = st.columns(len(metrics))
for col, (name, row) in zip(cols, metrics.iterrows()):
    col.metric(
        name,
        f"P&L: {row['final P&L']:.2f}",
        f"DD: {row['max drawdown']:.2f}  |  trades: {int(row['trades'])}",
        delta_color="off",
    )

st.dataframe(metrics.style.format({
    "final P&L": "{:.2f}",
    "max drawdown": "{:.2f}",
    "trades": "{:.0f}",
    "% time in market": "{:.1f}%",
}), use_container_width=True)


# ----- Charts ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Hedge ratio γₜ",
    "Spread + z-score",
    "Trading signal",
    "Cumulative return",
])

colors = {
    "Rolling LS": "#1f77b4",
    "Kalman basic": "#ff7f0e",
    "Kalman + momentum": "#2ca02c",
    "Kalman partial coint": "#d62728",
}


with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("γₜ (hedge ratio)", "μₜ (intercept)"))
    for name, series in res["gamma"].items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, line=dict(color=colors.get(name), width=1.2)), row=1, col=1)
    for name, series in res["mu"].items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, line=dict(color=colors.get(name), width=1.2), showlegend=False), row=2, col=1)
    fig.update_layout(height=620, legend=dict(orientation="h", y=1.07), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Reading it:** the rolling least-squares line jumps with each new "
        "window. The Kalman estimates smoothly track the underlying "
        "hedge ratio while still adapting to regime changes. Slide "
        "`alpha` down to make Kalman even smoother (but slower)."
    )


with tab2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Normalized spread zₜ", "Rolling z-score"))
    for name, series in res["spread"].items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, line=dict(color=colors.get(name), width=0.8)), row=1, col=1)
        z = spread_zscore(series, window=zwindow)
        fig.add_trace(go.Scatter(x=z.index, y=z.values, name=name + " (z)", line=dict(color=colors.get(name), width=0.8), showlegend=False), row=2, col=1)
    fig.add_hline(y=s0, line_dash="dot", row=2, col=1, line_color="grey")
    fig.add_hline(y=-s0, line_dash="dot", row=2, col=1, line_color="grey")
    fig.add_hline(y=0.0, line_color="black", line_width=1, row=2, col=1)
    fig.update_layout(height=620, legend=dict(orientation="h", y=1.07), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Strategy:** open long when z < -{s0}, short when z > +{s0}, "
        f"close at z = 0. Rolling z-score uses a {zwindow}-day window."
    )


with tab3:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Spread (Kalman basic)", "Position"))
    bt = backtests["Kalman basic"]
    spread = res["spread"]["Kalman basic"]
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="spread", line=dict(color="#888", width=0.7)), row=1, col=1)
    long_mask = bt["signal"] == 1
    short_mask = bt["signal"] == -1
    fig.add_trace(go.Scatter(x=spread.index[long_mask], y=spread.values[long_mask], mode="markers", name="long",
                              marker=dict(color="green", size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread.index[short_mask], y=spread.values[short_mask], mode="markers", name="short",
                              marker=dict(color="red", size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["signal"].values, name="position", line=dict(color="purple", shape="hv")), row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Green dots = long the spread (long y1 / short y2 × γ). "
        "Red dots = short the spread. Flat = waiting for the next entry."
    )


with tab4:
    fig = go.Figure()
    for name, bt in backtests.items():
        cum = bt["cumulative_return"]
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, line=dict(color=colors.get(name), width=1.6)))
    # Drawdown shading for the best performer
    best = max(backtests.items(), key=lambda kv: kv[1]["cumulative_return"].iloc[-1])
    cum = best[1]["cumulative_return"]
    dd = cum - cum.cummax()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=f"{best[0]} drawdown", fill="tozeroy",
                              line=dict(color=colors.get(best[0]), width=0), opacity=0.15))
    fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h", y=1.05),
                       title="Cumulative return (spread units, no transaction costs)")
    st.plotly_chart(fig, use_container_width=True)


# ----- Step-through animation (collapsed by default) ---------------------

with st.expander("Step-through: how the filter learns over time"):
    st.markdown("Drag the slider to see the Kalman filter incrementally process observations.")
    step = st.slider(
        "Up to day",
        min_value=t_ls + 50,
        max_value=len(prices) - 1,
        value=min(t_ls + 500, len(prices) - 1),
        step=20,
    )
    gamma_kb = res["gamma"]["Kalman basic"]
    spread_kb = res["spread"]["Kalman basic"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("γₜ learned so far", "Spread so far"))
    fig.add_trace(go.Scatter(x=gamma_kb.index[:step], y=gamma_kb.values[:step], name="γ"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread_kb.index[:step], y=spread_kb.values[:step], name="z"), row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.caption(
    "Built from `kalman_pairs.kalman` (Durbin & Koopman 2012 recursions). "
    "Real prices come from Yahoo Finance when available, otherwise from a "
    "deterministic synthetic generator."
)
