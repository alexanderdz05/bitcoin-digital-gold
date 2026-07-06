import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_asset_prices,
    ann_return,
    ann_vol,
    sharpe,
    max_dd_from_prices,
    CORE_ASSETS,
    ASSET_COLORS,
)

def show_test1():
    st.title("Test 1 — Gold-like Behavior")
    st.markdown("### Does Bitcoin actually behave like gold?")

    prices, returns = load_asset_prices()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Test 1 Controls")

    min_date = prices.index.min().date()
    max_date = prices.index.max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="test1_date_range",
    )

    if len(date_range) == 2:
        start, end = str(date_range[0]), str(date_range[1])
    else:
        start, end = str(min_date), str(max_date)

    prices_f = prices.loc[start:end, CORE_ASSETS]
    returns_f = returns.loc[start:end, CORE_ASSETS]

    if len(returns_f) < 60:
        st.warning("Please select a wider date range.")
        return

    st.info(
        "**Hypothesis:** If Bitcoin is digital gold, it should show gold-like behavior: "
        "lower volatility, smaller drawdowns, and meaningful correlation with Gold."
    )

    metrics = pd.DataFrame(index=CORE_ASSETS)
    metrics["Annualized Return"] = returns_f.apply(ann_return)
    metrics["Annualized Volatility"] = returns_f.apply(ann_vol)
    metrics["Sharpe Ratio"] = returns_f.apply(sharpe)
    metrics["Max Drawdown"] = prices_f.apply(max_dd_from_prices)

    st.subheader("Performance Metrics")

    display_metrics = metrics.copy()
    display_metrics["Annualized Return"] = display_metrics["Annualized Return"].map("{:.1%}".format)
    display_metrics["Annualized Volatility"] = display_metrics["Annualized Volatility"].map("{:.1%}".format)
    display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].map("{:.2f}".format)
    display_metrics["Max Drawdown"] = display_metrics["Max Drawdown"].map("{:.1%}".format)

    st.dataframe(display_metrics, use_container_width=True)

    st.divider()

    st.subheader("Correlation Matrix")

    corr = returns_f.corr()

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )

    fig_corr.update_layout(
        title="Return Correlation Matrix",
        height=430,
        template="plotly_white",
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    st.subheader("Historical Drawdowns")

    drawdowns = pd.DataFrame(index=prices_f.index)

    for asset in CORE_ASSETS:
        running_max = prices_f[asset].cummax()
        drawdowns[asset] = prices_f[asset] / running_max - 1

    fig_dd = go.Figure()

    for asset in CORE_ASSETS:
        fig_dd.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[asset],
                mode="lines",
                name=asset,
                line=dict(color=ASSET_COLORS[asset], width=2),
            )
        )

    fig_dd.update_layout(
        title="Historical Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=430,
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    st.subheader("Rolling 1-Year Correlation")

    roll = 252
    roll_btc_gold = returns_f["BTC"].rolling(roll).corr(returns_f["Gold"])
    roll_btc_spy = returns_f["BTC"].rolling(roll).corr(returns_f["SPY"])

    fig_roll = go.Figure()

    fig_roll.add_trace(
        go.Scatter(
            x=roll_btc_gold.index,
            y=roll_btc_gold,
            mode="lines",
            name="BTC vs Gold",
            line=dict(color=ASSET_COLORS["Gold"], width=2.5),
        )
    )

    fig_roll.add_trace(
        go.Scatter(
            x=roll_btc_spy.index,
            y=roll_btc_spy,
            mode="lines",
            name="BTC vs SPY",
            line=dict(color=ASSET_COLORS["SPY"], width=2.5),
        )
    )

    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_roll.update_layout(
        title="252-Day Rolling Correlation",
        xaxis_title="Date",
        yaxis_title="Correlation",
        height=430,
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig_roll, use_container_width=True)

    st.info(
        f"**Key Finding (Test 1):** BTC-Gold correlation is **{corr.loc['BTC', 'Gold']:.3f}**, "
        f"while BTC-SPY correlation is **{corr.loc['BTC', 'SPY']:.3f}**. "
        "Bitcoin has much higher volatility and deeper drawdowns than Gold, so it does not behave like digital gold."
    )