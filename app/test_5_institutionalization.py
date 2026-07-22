import streamlit as st
import pandas as pd
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

def show_test5():
    # Title and Header
    st.title("Test 5 — Post-2020 Institutionalization")
    st.markdown("### Did institutional adoption make Bitcoin more like gold?")

    # Pre vs Post 2020 Summary
    prices, returns = load_asset_prices()

    st.info(
        "**Hypothesis:** If institutional adoption strengthened the digital gold narrative, "
        "Bitcoin should become more gold-like after 2020. If it became more financialized, "
        "it may become more correlated with equities."
    )

    pre_prices = prices.loc[: "2020-12-31", CORE_ASSETS]
    pre_returns = returns.loc[: "2020-12-31", CORE_ASSETS]

    post_prices = prices.loc["2021-01-01" :, CORE_ASSETS]
    post_returns = returns.loc["2021-01-01" :, CORE_ASSETS]

    def period_metrics(price_data, return_data):
        df = pd.DataFrame(index=CORE_ASSETS)
        df["Annualized Return"] = return_data.apply(ann_return)
        df["Annualized Volatility"] = return_data.apply(ann_vol)
        df["Sharpe Ratio"] = return_data.apply(sharpe)
        df["Max Drawdown"] = price_data.apply(max_dd_from_prices)
        return df

    pre_metrics = period_metrics(pre_prices, pre_returns)
    post_metrics = period_metrics(post_prices, post_returns)

    pre_corr = pre_returns.corr()
    post_corr = post_returns.corr()

    st.subheader("Pre-Institutional vs Post-Institutional Summary")
    st.caption(
        "Split at Jan 1 2021 — MicroStrategy's August 2020 purchase began the institutional wave; "
        "the bulk of inflows (Tesla, Coinbase IPO, futures ETF, spot ETFs) arrived from 2021 onward."
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "BTC Annualized Return",
                "BTC Annualized Volatility",
                "BTC Max Drawdown",
                "BTC-SPY Correlation",
                "BTC-Gold Correlation",
            ],
            "Pre-Institutional (≤2020)": [
                pre_metrics.loc["BTC", "Annualized Return"],
                pre_metrics.loc["BTC", "Annualized Volatility"],
                pre_metrics.loc["BTC", "Max Drawdown"],
                pre_corr.loc["BTC", "SPY"],
                pre_corr.loc["BTC", "Gold"],
            ],
            "Post-Institutional (2021+)": [
                post_metrics.loc["BTC", "Annualized Return"],
                post_metrics.loc["BTC", "Annualized Volatility"],
                post_metrics.loc["BTC", "Max Drawdown"],
                post_corr.loc["BTC", "SPY"],
                post_corr.loc["BTC", "Gold"],
            ],
        }
    )

    display_summary = summary.copy()

    for idx in [0, 1, 2]:
        display_summary.loc[idx, "Pre-Institutional (≤2020)"] = f"{summary.loc[idx, 'Pre-Institutional (≤2020)']:.1%}"
        display_summary.loc[idx, "Post-Institutional (2021+)"] = f"{summary.loc[idx, 'Post-Institutional (2021+)']:.1%}"

    for idx in [3, 4]:
        display_summary.loc[idx, "Pre-Institutional (≤2020)"] = f"{summary.loc[idx, 'Pre-Institutional (≤2020)']:.3f}"
        display_summary.loc[idx, "Post-Institutional (2021+)"] = f"{summary.loc[idx, 'Post-Institutional (2021+)']:.3f}"

    st.dataframe(display_summary, hide_index=True, use_container_width=True)

    st.divider()

    # Rolling Correlation Around Institutionalization
    st.subheader("Rolling Correlation: BTC vs SPY and Gold (252-Day Window)")

    roll = 252

    roll_btc_spy = returns["BTC"].rolling(roll).corr(returns["SPY"])
    roll_btc_gold = returns["BTC"].rolling(roll).corr(returns["Gold"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=roll_btc_spy.index,
            y=roll_btc_spy,
            mode="lines",
            name="BTC vs SPY",
            line=dict(color="#DC2626", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=roll_btc_gold.index,
            y=roll_btc_gold,
            mode="lines",
            name="BTC vs Gold",
            line=dict(color=ASSET_COLORS["Gold"], width=2.5),
        )
    )

    # Shaded region starts Aug 11 2020 — MicroStrategy's first purchase,
    # the commonly-cited start of the institutional adoption wave
    fig.add_vrect(
        x0="2020-08-11",
        x1=str(prices.index.max().date()),
        fillcolor="rgba(99,102,241,0.08)",
        line_width=0,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="252-Day Rolling Correlation: BTC vs SPY and Gold",
        xaxis_title="Date",
        yaxis_title="Correlation",
        height=480,
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**Key Finding (Test 5):** BTC-SPY correlation increased from "
        f"**{pre_corr.loc['BTC', 'SPY']:.3f}** (pre-institutional) to **{post_corr.loc['BTC', 'SPY']:.3f}** (post-institutional). "
        f"BTC-Gold correlation remained weak at **{post_corr.loc['BTC', 'Gold']:.3f}**. "
        "The institutional wave — beginning with MicroStrategy in August 2020 and accelerating through Tesla, "
        "futures ETFs (Oct 2021), and spot ETFs (Jan 2024) — made Bitcoin more correlated with equities, not more gold-like."
    )