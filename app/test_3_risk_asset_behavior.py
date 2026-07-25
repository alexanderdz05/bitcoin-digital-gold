import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    load_asset_prices,
    ann_return,
    ann_vol,
    sharpe,
    CORE_ASSETS,
    ASSET_COLORS,
)

def show_test3():
    # Title and Header
    st.title("Test 3 — Risk Asset Behavior")
    st.markdown("### Does Bitcoin behave like a high-beta risk asset?")

    # Risk on vs Risk off Analysis
    prices, returns = load_asset_prices()

    if "VIX" not in prices.columns:
        st.error("VIX column not found in asset_prices.csv.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risk Regime Controls")

    min_date = prices.index.min().date()
    max_date = prices.index.max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="test3_date_range",
    )

    vix_threshold = st.sidebar.slider(
        "VIX Stress Threshold",
        min_value=10,
        max_value=60,
        value=20,
        step=1,
        key="test3_vix_threshold",
    )

    if len(date_range) == 2:
        start, end = str(date_range[0]), str(date_range[1])
    else:
        start, end = str(min_date), str(max_date)

    prices_f = prices.loc[start:end]
    returns_f = returns.loc[start:end]

    if len(returns_f) < 60:
        st.warning("Please select a wider date range.")
        return

    st.info(
        "**Hypothesis:** If Bitcoin is digital gold, it should hold up during risk-off markets. "
        "If it is a risk asset, it should perform better when VIX is low and worse when VIX is elevated."
    )

    vix_levels = prices_f["VIX"]

    risk_on = returns_f[vix_levels <= vix_threshold]
    risk_off = returns_f[vix_levels > vix_threshold]

    st.subheader(f"VIX Regime Breakdown (Threshold: {vix_threshold})")

    def make_regime_table(regime_returns):
        if len(regime_returns) < 10:
            return None

        return pd.DataFrame(
            {
                "Asset": CORE_ASSETS,
                "Ann. Return": [f"{ann_return(regime_returns[a]):.1%}" for a in CORE_ASSETS],
                "Volatility": [f"{ann_vol(regime_returns[a]):.1%}" for a in CORE_ASSETS],
                "Sharpe": [f"{sharpe(regime_returns[a]):.2f}" for a in CORE_ASSETS],
            }
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Risk-On (VIX ≤ {vix_threshold}) — {len(risk_on)} days**")
        table = make_regime_table(risk_on)
        if table is not None:
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("Insufficient data.")

    with col2:
        st.markdown(f"**Risk-Off (VIX > {vix_threshold}) — {len(risk_off)} days**")
        table = make_regime_table(risk_off)
        if table is not None:
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("Insufficient data.")

    st.divider()

    # Annualized Returns by Regime
    if len(risk_on) > 10 and len(risk_off) > 10:
        fig = go.Figure()

        regimes = [
            f"Risk-On (VIX≤{vix_threshold})",
            f"Risk-Off (VIX>{vix_threshold})",
        ]

        for asset in CORE_ASSETS:
            fig.add_trace(
                go.Bar(
                    name=asset,
                    x=regimes,
                    y=[
                        ann_return(risk_on[asset]),
                        ann_return(risk_off[asset]),
                    ],
                    marker_color=ASSET_COLORS[asset],
                )
            )

        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            title="Annualized Returns by VIX Regime",
            yaxis_title="Annualized Return",
            yaxis_tickformat=".0%",
            barmode="group",
            height=420,
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key Finding (Test 3):** "
        "Bitcoin performs best during risk-on markets and weakens during stress, supporting the high-beta risk asset view."
    )