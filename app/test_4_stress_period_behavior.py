import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

from utils import (
    load_asset_prices,
    cumulative_return,
    max_dd_from_prices,
    recovery_days_from_trough,
    STRESS_PERIODS,
    CORE_ASSETS,
    ASSET_COLORS,
)

def show_test4():
    st.title("Test 4 — Stress-Period Behavior")
    st.markdown("### Does Bitcoin protect capital during real market crises?")

    prices, returns = load_asset_prices()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Stress Periods")

    selected_periods = st.sidebar.multiselect(
        "Select Periods",
        options=list(STRESS_PERIODS.keys()),
        default=[
            "COVID Crash (Feb–Mar 2020)",
            "2022 Inflation & Rate Hikes",
        ],
        key="test4_periods",
    )

    st.sidebar.markdown("**Custom Period**")

    custom_label = st.sidebar.text_input(
        "Label (optional)",
        value="",
        key="test4_custom_label",
    )

    st.info(
        "**Hypothesis:** If Bitcoin is digital gold, it should preserve value during crisis periods "
        "more like Gold than SPY."
    )

    active_periods = {k: STRESS_PERIODS[k] for k in selected_periods}

    if not active_periods:
        st.warning("Select at least one stress period.")
        return

    stress_rows = []

    for period_name, (start, end) in active_periods.items():
        p_prices = prices.loc[start:end, CORE_ASSETS]
        p_returns = returns.loc[start:end, CORE_ASSETS]

        if len(p_returns) < 5:
            st.warning(f"Not enough data for {period_name}. Skipping.")
            continue

        for asset in CORE_ASSETS:
            stress_rows.append(
                {
                    "Period": period_name,
                    "Asset": asset,
                    "Cumulative Return": cumulative_return(p_returns[asset]),
                    "Max Drawdown": max_dd_from_prices(p_prices[asset]),
                    "Recovery Days": recovery_days_from_trough(p_prices[asset]),
                }
            )

    stress_df = pd.DataFrame(stress_rows)

    if stress_df.empty:
        st.warning("No valid stress period data.")
        return

    period_order = list(active_periods.keys())

    st.subheader("Cumulative Returns During Stress Periods")

    fig = go.Figure()

    for asset in CORE_ASSETS:
        asset_rows = stress_df[stress_df["Asset"] == asset].set_index("Period")
        y_vals = [
            asset_rows.loc[p, "Cumulative Return"] if p in asset_rows.index else None
            for p in period_order
        ]

        fig.add_trace(
            go.Bar(
                name=asset,
                x=period_order,
                y=y_vals,
                marker_color=ASSET_COLORS[asset],
            )
        )

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title="Cumulative Returns During Stress Periods",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        barmode="group",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress Period Detail")

    def pivot_metric(metric):
        return (
            stress_df.pivot(index="Period", columns="Asset", values=metric)
            .reindex(period_order)[CORE_ASSETS]
        )

    tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Max Drawdowns", "Recovery Days"])

    with tab1:
        st.dataframe(pivot_metric("Cumulative Return").style.format("{:.1%}"), use_container_width=True)

    with tab2:
        st.dataframe(pivot_metric("Max Drawdown").style.format("{:.1%}"), use_container_width=True)

    with tab3:
        rec = pivot_metric("Recovery Days")

        def fmt_rec(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "Not Recovered"
            return f"{int(x)} days"

        st.dataframe(rec.map(fmt_rec), use_container_width=True)

    st.divider()

    st.subheader("Normalized Price Chart")

    drill_period = st.selectbox(
        "Select Period to Drill Into",
        options=period_order,
        key="test4_drill",
    )

    d_start, d_end = active_periods[drill_period]
    drill_prices = prices.loc[d_start:d_end, CORE_ASSETS]

    if len(drill_prices) < 2:
        st.warning("Not enough data.")
        return

    normalized = drill_prices / drill_prices.iloc[0] * 100

    fig_norm = go.Figure()

    for asset in CORE_ASSETS:
        fig_norm.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[asset],
                mode="lines",
                name=asset,
                line=dict(color=ASSET_COLORS[asset], width=2.5),
            )
        )

    fig_norm.add_hline(
        y=100,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Starting value (100)",
    )

    fig_norm.update_layout(
        title=f"Normalized Price — {drill_period}",
        xaxis_title="Date",
        yaxis_title="Indexed Price",
        height=430,
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig_norm, use_container_width=True)

    st.info(
        "**Key Finding (Test 4):** Bitcoin lost significantly more than Gold during major stress periods. "
        "This behavior is consistent with a high-beta risk asset, not a safe haven."
    )